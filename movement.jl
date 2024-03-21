function move!(particle, species, mesh, boundaries, time_step)
    done = false
    while !done
        particle.position += time_step * particle.velocity[1]

        bc = 0
        if particle.position < limits(mesh)[1]
            bc = 1
        elseif particle.position > limits(mesh)[2]
            bc = 2
        else
            done = true
        end

        if !iszero(bc)
            # Move particle to boundary
            time_step -= (particle.position - limits(mesh)[bc]) / particle.velocity[1]
            particle.position = limits(mesh)[bc]

            collide!(particle, boundaries[bc]; species)
            particle.velocity = particle.velocity .* (3 - bc * 2, 1, 1)
        end
    end
end